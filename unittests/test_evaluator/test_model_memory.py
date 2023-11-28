

import torch
import unittest

from evaluator.memory_usage import get_max_memory_usage, print_memory


class TestMemoryUsageEvaluation(unittest.TestCase):
    def test_get_max_memory_usage(self):
        if not torch.cuda.is_available():
            print("Skipping test_get_max_memory_usage on CPU.")
            return

        def gpu_matmul():
            m1, m2 = torch.randn(256, 256).cuda(), torch.randn(256, 256).cuda()
            return torch.matmul(m1, m2)

        print_memory("Start")
        mm = get_max_memory_usage(target_function=gpu_matmul)
        print_memory("After Matmul")

        # Expected: 3 Matrices, size 256x256, of float32, 8 bits per byte
        expected_mm = 3 * (256 * 256) * 32 // 8
        self.assertEqual(mm, expected_mm)

    # def test_model_memory_usage(self):
    #     bs = 256
    #     ds = 10 ** 6
    #     ins = 32 * 32
    #     outs = 100
    #
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()
    #
    #     print("x" * 100)
    #     print_memory("Start")
    #
    #     dummy_model = nn.Sequential(
    #         nn.Linear(ins, outs),
    #         nn.Softmax(dim=1),
    #     )
    #
    #     print_memory("After defining model")
    #
    #     dummy_loader = torch.utils.data.DataLoader(
    #         dataset=torch.utils.data.TensorDataset(torch.ones((ds, ins)),
    #                                                torch.zeros((ds, ins))),
    #         batch_size=bs,
    #     )
    #     dummy_optimizer = torch.optim.SGD(dummy_model.parameters(),
    #                                       lr=0.01, momentum=0.9)
    #
    #     print_memory("After defining dataset")
    #
    #     print("\n\nExpected:")
    #     expectations = {
    #         "Model size": (ins * outs) * 32 // 8,
    #         "Gradient": (ins * outs) * 32 // 8,
    #         "Batch x": bs * ins * 32 // 8,
    #         "Forward": bs * (ins + outs) * 32 // 8,  # Includes batch x
    #         "Backward": bs * (ins + outs) * 32 // 8,  # + some extra
    #         "SGD+mom": (ins * outs) * 32 // 8,
    #         "Output": bs * outs * 32 // 8
    #     }
    #     print("\n".join(
    #         f"{key}: {v:,}" for key, v in expectations.items()) + "\n\n")
    #
    #     backward = functools.partial(backward_pass,
    #                                  model=dummy_model,
    #                                  data_loader=dummy_loader,
    #                                  optimizer=dummy_optimizer,
    #                                  device="cuda")
    #     train_mm = get_max_memory_usage(target_function=backward)
    #     print(f"Found max memory of {train_mm} for training.")
    #     expected_keys = ["Model size", "Gradient", "Forward", "SGD+mom"]
    #     expected_mm = sum(expectations[key] for key in expected_keys)
    #     print(f"Train max memory expected to be (about) {expected_mm:,}.")
    #     print_memory("after get_max_memory_usage(backward)")
    #     temp_mem_bound = expectations["Model size"] + expectations["Output"]
    #     assert expected_mm <= train_mm <= 1.1 * (expected_mm + temp_mem_bound), \
    #         f"Error: {expected_mm:,} <= {train_mm:,} " \
    #         f"<= 1.1 * {expected_mm + temp_mem_bound:,} does not hold."
    #
    #     # Empty memory to measure forward pass memory usage:
    #     del backward
    #     del dummy_model
    #     del dummy_optimizer
    #     torch.cuda.reset_peak_memory_stats()
    #
    #     dummy_model = nn.Sequential(
    #         nn.Linear(ins, outs),
    #         nn.Softmax(dim=1),
    #     )
    #
    #     print("x" * 100)
    #     print_memory("Start of forward")
    #     forward = functools.partial(forward_pass,
    #                                 model=dummy_model,
    #                                 data_loader=dummy_loader,
    #                                 device="cuda")
    #     val_mm = get_max_memory_usage(function=forward)
    #     expected_mm = expectations["Model size"] + expectations["Forward"]
    #     print(f"Val mm should be (about) {expected_mm:,}")
    #     print_memory()
    #     temp_mem_bound = expectations["Model size"]
    #     assert expected_mm <= val_mm <= 1.1 * (expected_mm + temp_mem_bound), \
    #         f"Error: {expected_mm:,} <= {val_mm:,} " \
    #         f"<= 1.1 * {expected_mm + temp_mem_bound:,} does not hold."
    #
    #     # cached_forward = torch.nn.utils.parametrize.cached()(forward)
    #     # val_mm = get_max_memory_usage(function=cached_forward)
    #     # print(val_mm)


