import torch


class Logger:
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            results = 100 * torch.tensor(self.results[run])
            argmax = results[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {results[:, 0].max():.2f}")
            print(f"Highest Valid: {results[:, 1].max():.2f}")
            print(f"  Final Train: {results[argmax, 0]:.2f}")
            print(f"   Final Test: {results[argmax, 2]:.2f}")
        else:
            results = 100 * torch.tensor(self.results)

            best_results = []
            for result in results:
                train1 = result[:, 0].max().item()
                valid = result[:, 1].max().item()
                train2 = result[result[:, 1].argmax(), 0].item()
                test = result[result[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print("All runs:")
            result = best_result[:, 0]
            print(f"Highest Train: {result.mean():.2f} ± {result.std():.2f}")
            result = best_result[:, 1]
            print(f"Highest Valid: {result.mean():.2f} ± {result.std():.2f}")
            result = best_result[:, 2]
            print(f"  Final Train: {result.mean():.2f} ± {result.std():.2f}")
            result = best_result[:, 3]
            print(f"   Final Test: {result.mean():.2f} ± {result.std():.2f}")
