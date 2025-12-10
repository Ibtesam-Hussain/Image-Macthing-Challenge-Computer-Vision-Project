from src.pipeline import main, parse_args


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

