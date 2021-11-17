


def main(num):
    bin_ = list(bin(num)[2:])
    idx_list = [i for i, x in enumerate(bin_) if x =='1']
    max = 0
    for idx_, idx in enumerate(idx_list[1:]):
        sub = (idx - idx_list[idx_] - 1)
        if sub > max:
            max = sub

main(1041)