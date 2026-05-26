"""WBR 回测统一入口：根据 --mode 分发到 single 或 ga_run"""
import sys


def main():
    args = sys.argv[1:]
    mode = 'ga'
    for i, a in enumerate(args):
        if a == '--mode' and i + 1 < len(args):
            mode = args[i + 1]
            break

    if mode == 'single':
        from testback.single import main as _main
    else:
        from testback.ga_run import main as _main

    _main()


if __name__ == '__main__':
    main()
