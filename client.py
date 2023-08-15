import argparse
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def main(prompt: str):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    text_obj = np.array([prompt], dtype="object")
    
    input_tensors = [
        httpclient.InferInput("TEXT", text_obj.shape, np_to_triton_dtype(text_obj.dtype)),
    ]

    input_tensors[0].set_data_from_numpy(text_obj)


    output = [
        httpclient.InferRequestedOutput("TEXT_OUT"),
    ]

    # Query
    query_response = client.infer(model_name="ensemble_model",
                                  inputs=input_tensors,
                                  outputs=output)

    print(query_response.as_numpy('TEXT_OUT')[0].decode("UTF-8"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the answer to life, the universe, and everything?")
    args = parser.parse_args()
    main(args.prompt)