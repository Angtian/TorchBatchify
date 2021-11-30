import torch
import threading
from collections.abc import Iterable
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper
from typing import Union


class Batchifier(object):
    def __init__(self, batch_size: int, batch_args: Union[str, list, tuple], target_dims: Union[int, list, tuple, None] = None, remain_dims: Union[int, list, tuple, None] = None):
        """
        Automatically batchify a process. remain_dims must be placed at start and end of the shape.
        :param batch_size: batch size
        :param batch_args: name of args to batchify
        :param target_dims: the raveled dims
        :param remain_dims: the unraveled dims
        """
        if isinstance(batch_args, str):
            batch_args = (batch_args, )

        if target_dims is not None:
            if isinstance(target_dims, int):
                target_dims = (target_dims, )
            self.target_dims = tuple(target_dims)
            self.remain_dims = None
        else:
            if isinstance(remain_dims, int):
                remain_dims = (remain_dims, )
            self.remain_dims = tuple(remain_dims)
            self.target_dims = None

        self.batch_size = batch_size
        self.batch_args = tuple(batch_args)

        assert len(self.batch_args) > 0

    def __call__(self, func):
        """
        Function decorator for Batchifier.
        :param func: a callable function or object (e.g. torch.nn.functional)
        :return: the batchified function
        """
        def wrapper(*args, **kwargs):
            kwargs = dict(kwargs)

            total_len = -1

            recorded_shape = None
            save_idx = None
            for k in self.batch_args:
                get = kwargs[k]

                assert isinstance(get, torch.Tensor)

                if self.target_dims is not None:
                    this_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.target_dims])
                    other_dims = tuple([i for i in range(len(get.shape)) if i not in this_dims])
                else:
                    other_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.remain_dims])
                    this_dims = tuple([i for i in range(len(get.shape)) if i not in other_dims])

                to_shape = [get.shape[i] if i in other_dims else -1 for i in range(len(get.shape))]
                t_l = len(to_shape)

                for i in range(t_l - 1):
                    if to_shape[t_l - 1 - i] == -1 and to_shape[t_l - 2 - i] == -1:
                        del to_shape[t_l - 1 - i]

                assert to_shape.count(-1) == 1

                to_record_shape = get.shape[0:to_shape.index(-1) + len(this_dims)]
                if recorded_shape is None:
                    recorded_shape = tuple(to_record_shape)
                    save_idx = to_shape.index(-1)
                else:
                    assert recorded_shape == tuple(to_record_shape)

                kwargs[k] = get.view(*to_shape)

                total_len = kwargs[k].shape[to_shape.index(-1)]

            assert total_len >= 0, 'No batchify parameters found!'

            reshape_foo = lambda x_, tar_shape=recorded_shape: x_.view(*tar_shape + x_.shape[save_idx + 1:])

            out = []
            for i in range((total_len - 1) // self.batch_size + 1):
                this_kwargs = dict()
                for k in kwargs.keys():
                    this_kwargs[k] = kwargs[k]
                    if k in self.batch_args:
                        exec('this_kwargs[k] = this_kwargs[k][%si * self.batch_size: (i + 1) * self.batch_size]' % ''.join((':, ', ) * save_idx))

                out.append(func(*args, **this_kwargs))
            
            if isinstance(out[0], tuple):
                out_reshaped = [[] for _ in range(len(out[0]))]
                for this_out in out:
                    for i in range(len(out[0])):
                        out_reshaped[i].append(this_out[i])

                return tuple([reshape_foo(torch.cat(this_row, dim=save_idx)) for this_row in out_reshaped])
            else:
                return reshape_foo(torch.cat(out, dim=save_idx))

        return wrapper


class DataParallelBatchifier(object):
    def __init__(self, batch_size: int, batch_args: Union[str, list, tuple], target_dims: Union[int, list, tuple, None] = None, remain_dims: Union[int, list, tuple, None] = None, device: torch.device=None):
        """
        Automatically batchify a process with mutliple GPUs. remain_dims must be placed at start and end of the shape.
        :param batch_size: batch size
        :param batch_args: name of args to batchify
        :param target_dims: the raveled dims
        :param remain_dims: the unraveled dims
        :param device: device to use, use all gpu if None
        """
        if isinstance(batch_args, str):
            batch_args = (batch_args, )

        if target_dims is not None:
            if isinstance(target_dims, int):
                target_dims = (target_dims, )
            self.target_dims = tuple(target_dims)
            self.remain_dims = None
        else:
            if isinstance(remain_dims, int):
                remain_dims = (remain_dims, )
            self.remain_dims = tuple(remain_dims)
            self.target_dims = None

        if device is None:
            self.device = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        else:
            assert isinstance(device, Iterable), 'Device must be iterable, get %s.' % str(device)
            self.device = device

        self.n_gpus = len(self.device)

        self.batch_size = batch_size
        self.batch_args = tuple(batch_args)

        assert len(self.batch_args) > 0

    def __call__(self, func):
        """
        Function decorator for Batchifier.
        :param func: a callable function or object (e.g. torch.nn.functional)
        :return: the batchified function
        """
        def _worker(lock, foo, results, device_idx, this_device, args, kwargs, autocast_enabled):
            try:
                with torch.cuda.device(this_device), autocast(enabled=autocast_enabled):
                    # this also avoids accidental slicing of `input` if it is a Tensor
                    output = foo(*args, **kwargs)
                with lock:
                    results[device_idx] = output
            except Exception:
                with lock:
                    results[device_idx] = ExceptionWrapper(
                        where="in replica {} on device {}".format(device_idx, this_device))

        def wrapper(*args, **kwargs):
            kwargs = dict(kwargs)

            total_len = -1

            recorded_shape = None
            save_idx = None
            current_device = None
            lock = threading.Lock()
            grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

            for k in self.batch_args:
                get = kwargs[k]

                assert isinstance(get, torch.Tensor)

                if current_device is None:
                    current_device = get.device
                else:
                    assert current_device == get.device, 'args must be on same device, but get %s, %s' % (str(current_device), str(get.device))

                if self.target_dims is not None:
                    this_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.target_dims])
                    other_dims = tuple([i for i in range(len(get.shape)) if i not in this_dims])
                else:
                    other_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.remain_dims])
                    this_dims = tuple([i for i in range(len(get.shape)) if i not in other_dims])

                to_shape = [get.shape[i] if i in other_dims else -1 for i in range(len(get.shape))]
                t_l = len(to_shape)

                for i in range(t_l - 1):
                    if to_shape[t_l - 1 - i] == -1 and to_shape[t_l - 2 - i] == -1:
                        del to_shape[t_l - 1 - i]

                assert to_shape.count(-1) == 1

                to_record_shape = get.shape[0:to_shape.index(-1) + len(this_dims)]
                if recorded_shape is None:
                    recorded_shape = tuple(to_record_shape)
                    save_idx = to_shape.index(-1)
                else:
                    assert recorded_shape == tuple(to_record_shape)

                kwargs[k] = get.view(*to_shape)

                total_len = kwargs[k].shape[to_shape.index(-1)]

            assert total_len >= 0, 'No batchify parameters found!'

            reshape_foo = lambda x_, tar_shape=recorded_shape: x_.view(*tar_shape + x_.shape[save_idx + 1:])

            out = []
            for i in range((total_len - 1) // self.batch_size + 1):
                sub_batch_size = (self.batch_size - 1) // self.n_gpus + 1 if i != total_len // self.batch_size else (total_len % self.batch_size - 1) // self.n_gpus + 1

                results = dict()
                threads = []
                for j, this_device in enumerate(self.device):
                    this_kwargs = dict()
                    for k in kwargs.keys():
                        if k in self.batch_args:
                            start_idx = i * self.batch_size + min(j * sub_batch_size, self.batch_size)
                            end_idx = i * self.batch_size + min((j + 1) * sub_batch_size, self.batch_size)
                            exec('this_kwargs[k] = kwargs[k][%s start_idx: end_idx]' % ''.join((':, ', ) * save_idx))
                        else:
                            this_kwargs[k] = kwargs[k]
                        if torch.is_tensor(this_kwargs[k]):
                            this_kwargs[k] = this_kwargs[k].to(this_device)

                    if this_kwargs[self.batch_args[0]].shape[save_idx] != 0:
                        threads.append(threading.Thread(target=_worker, args=(lock, func, results, j, this_device, args, this_kwargs, autocast_enabled)))

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                
                for j, _ in enumerate(self.device):
                    if not j in results.keys():
                        continue
                    if isinstance(results[j], ExceptionWrapper):
                        results[j].reraise()
                    else:
                        this_out = results[j]
                        if torch.is_tensor(this_out):
                            out.append(this_out.to(current_device))
                        else:
                            out.append(tuple([t.to(current_device) for t in this_out]))

            if isinstance(out[0], tuple):
                out_reshaped = [[] for _ in range(len(out[0]))]
                for this_out in out:
                    for i in range(len(out[0])):
                        out_reshaped[i].append(this_out[i])

                return tuple([reshape_foo(torch.cat(this_row, dim=save_idx)) for this_row in out_reshaped])
            else:
                return reshape_foo(torch.cat(out, dim=save_idx))

        return wrapper

