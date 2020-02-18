import torch


def to_sorted_tensor(tensor, lengths, sort_dim, device):
    """
    Sort tensor according to sequence lengths.
    This is used before applying pack_padded_sequence.
    :param tensor: input tensor
    :param lengths: 1D tensor of sequence lengths
    :param sort_dim: the dimension to sort in input tensor
    :param device: calculation device
    :return: sorted tensor, sorted lengths, and sorted index.
    """
    sorted_lengths, sorted_idx = torch.sort(
        lengths.long(), dim=0, descending=True)
    # sorted_idx = sorted_idx.to(device)
    sorted_tensor = tensor.index_select(dim=sort_dim, index=sorted_idx)
    return sorted_tensor, sorted_lengths, sorted_idx


def to_original_tensor(sorted_tensor, sorted_idx, sort_dim, device):
    """
    Restore tensor to its original order.
    This is used after applying pad_packed_sequence.
    :param sorted_tensor: a sorted tensor
    :param sorted_idx: sorted index of the sorted_tensor
    :param sort_dim: the dimension of sorted_tensor where it is sorted
    :device: calculation device
    :return: the original unsorted tensor
    """
    original_idx = torch.LongTensor(sorted(
        range(sorted_idx.size(0)), key=lambda x: sorted_idx[x])).to(device)
    tensor = sorted_tensor.index_select(dim=sort_dim, index=original_idx)
    return tensor


def to_thinnest_padded_tensor(tensor, PAD=0):
    """
    Given an 2D padded tensor of shape (batch_size, padded_length),
    resize it to (batch_size, maximum_unpadded_length).
    :param tensor: input 2D tensor
    :param PAD: value of pad id
    :return: resized tensor, and the true maximum unpadded length.
    """
    mask = (torch.ones_like(tensor) * PAD != tensor).float()
    lengths = mask.sum(dim=1)
    max_len = int(lengths.max().item())
    resized_tensor = tensor[:, :max_len]
    return resized_tensor, max_len


def transform_tensor_by_dict(tensor, map_dict, device="cpu"):
    """
    Given a tensor of any shape, map each of its value
    to map_dict[value].
    :param tensor: input tensor
    :param map_dict: value mapping dict. Both key and value are numbers
    :return: mapped tensor, type is FloatTensor
    """
    mapped_tensor = torch.FloatTensor(
        [map_dict[k] for k in tensor.reshape(-1).cpu().tolist()]
    ).reshape(tensor.shape).to(device)
    return mapped_tensor


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


if __name__ == "__main__":
    # test transform_tensor_by_dict
    tensor = torch.LongTensor([[1, 2], [3, 4]])
    map_dict = {1: 11, 2: 22, 3: 33, 4: 44}
    print(transform_tensor_by_dict(tensor, map_dict).long())
