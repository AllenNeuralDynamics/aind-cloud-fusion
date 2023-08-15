import io
import fusion

def main():
    CHUNK_SIZE = (512, 512, 512)
    OUTPUT_LOCATION = ...

    dataset = io.load_dataset_from_big_stitcher(...)
    fusion.fuse(dataset, CHUNK_SIZE, OUTPUT_LOCATION)

if __name__ == '__main__':
    main()