import torch
from torch.nn import functional as F
def main():
    print("Hello from sdiff!")
    x = torch.ones(1, 1, 1, 1)
    print(x, end="\n\n")

    # # pad last dimension "at the end"
    # x1 = F.pad(x, (0, 0, 0, 0, 0, 0, 1, 0))
    # print(x1, end="\n\n")

    # # pad last dimension "at the front"
    # x2 = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
    # print(x2, end="\n\n")

    # # pad last dimension on both sides with two zeros
    # x3 = F.pad(x, (0, 0, 0, 0, 0, 0, 2, 2))
    # print(x3, end="\n\n")

    # # pad dim1 "at the front" with 4 values
    # x4 = F.pad(x, (0, 0, 4, 0, 0, 0, 0, 0))
    # print(x4, end="\n\n")

    x5 = F.pad(x, (0, 0, 0, 1))
    print(f'x5:{x5}', end="\n\n")

    x6 = F.pad(x, (0, 0, 0, 1, 0 ,0, 0, 0))
    print(f'x6:{x6}', end="\n\n")

    print(f"x5==x6: {torch.equal(x5, x6)}")

if __name__ == "__main__":
    main()
