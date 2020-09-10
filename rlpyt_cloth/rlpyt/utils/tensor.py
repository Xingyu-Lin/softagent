
import torch


def select_at_indexes(indexes, tensor):
    """Leading dimensions of tensor must match dimensions of indexes."""
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def to_onehot(indexes, num, dtype=None):
    """Dimension of size num added to the end of indexes.shape."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Selected dimension of onehot is removed by argmax."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(tensor, valid=None, dim=None):
    dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=dim)
    valid = valid.type(tensor.dtype)  # Convert as needed.
    return (tensor * valid).sum(dim=dim) / valid.sum(dim=dim)


def infer_leading_dims(tensor, dim):
    """Param 'dim': number of non-leading dimensions in tensor.
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[-dim:]
    return lead_dim, T, B, shape


def restore_leading_dims(tensors, lead_dim, T=1, B=1):
    """Assume tensors have leading Batch dimension (might need removed)."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]


def repeat(tensor, dims):
    if len(dims) != len(tensor.shape):
        raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
    for index, dim in enumerate(dims):
        repetition_vector = [1]*(len(dims)+1)
        repetition_vector[index+1] = dim
        new_tensor_shape = list(tensor.shape)
        new_tensor_shape[index] *= dim
        tensor = tensor.unsqueeze(index+1).repeat(repetition_vector).reshape(new_tensor_shape)
    return tensor


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)