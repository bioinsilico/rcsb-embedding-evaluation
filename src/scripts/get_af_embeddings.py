import argparse

from pymilvus import Collection, connections
import struct
import numpy as np
from typing import List, Any


def binary_to_float_np_array(binary_data):
    float16_list = []
    for i in range(0, len(binary_data), 2):
        try:
            float16_value = np.frombuffer(struct.pack('H', struct.unpack('<H', binary_data[i:i+2])[0]), dtype=np.float16)[0]
            float16_list.append(float16_value)
        except struct.error:
            print(f"Warning: Not enough data to unpack at index {i}. Skipping.")
            break
    return np.array(float16_list)


def batch_list(data: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def write_array_to_column_file(array, filename):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if array.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    np.savetxt(filename, array, fmt='%.6f')


def fetch_embeddings_by_ids(
        milvus_host: str,
        milvus_port: str,
        collection_name: str,
        id_list: list,
        vector_field_name = "embedding",
        id_field_name = "id"

):
    # Connect to Milvus
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    # Load the collection
    collection = Collection(name=collection_name)
    collection.load()


    # Perform the query
    query_exp = ",".join([f'"{i}"' for i in id_list])
    results = collection.query(
        expr=f'{id_field_name} in [{query_exp}]',
        output_fields=[id_field_name, vector_field_name]
    )

    results = [{
        'id': r['id'],
        'embedding': binary_to_float_np_array(r['embedding'][0])
    } for r in results]

    return results

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--milvus-url', type=str, required=True)
    parser.add_argument('--milvus-port', type=str, required=True)
    parser.add_argument('--collection-name', type=str, required=True)
    parser.add_argument('--query-file', type=str, required=True)
    parser.add_argument('--target-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    milvus_host = args.milvus_url
    milvus_port = args.milvus_port
    collection_name = args.collection_name

    query_list = [r for r in open(args.query_file, "r")]
    target_list = [r for r in open(args.target_file, "r")]
    id_list = ["-".join(r.split("-")[0:3]) for r in query_list + target_list]

    for ids in batch_list(id_list, 10):
        embeddings = fetch_embeddings_by_ids(
            milvus_host,
            milvus_port,
            collection_name,
            ids
        )
        for e in embeddings:
            write_array_to_column_file(e['embedding'], f"{args.output_folder}/{e['id']}.csv")
