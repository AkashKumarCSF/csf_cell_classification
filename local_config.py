import sacred.observers

log_dir_root = "test_data/out/runs"  # Root of logging directory


def setup_logger(ex):
    # Configuration of Sacred experiment logger
    pass


def add_observers(ex):
    ex.observers.append(
        sacred.observers.FileStorageObserver(log_dir_root)
    )
