"""Check shared inference-metric subtraction."""

from lits.lm import subtract_inference_metrics


def main() -> None:
    after = {
        "num_calls": 5,
        "input_tokens": 120,
        "output_tokens": 30,
        "running_time": 2.5,
    }
    before = {
        "num_calls": 2,
        "input_tokens": 40,
        "output_tokens": 10,
        "running_time": 1.0,
    }
    difference = subtract_inference_metrics(after, before)
    assert difference == {
        "num_calls": 3,
        "num_batch_calls": 0,
        "num_flatten_calls": 0,
        "input_tokens": 80,
        "output_tokens": 20,
        "running_time": 1.5,
    }
    print(f"Inference metric difference: {difference}")


if __name__ == "__main__":
    main()
