from typing import Callable, Iterable

import multiprocessing as mp
from multiprocessing.managers import DictProxy

class SequentialStreamingExecutor[I, O]:
    """
    Executor that streams results sequentially from a task.
    """

    def __init__(self, task: Callable[[I], tuple[I, O]], inputs: Iterable[I]):
        """
        Initializes the executor with a task to execute.

        Starts a multiprocessing pool and a manager to handle shared state.

        :param task: A callable that takes an integer (the task index) and returns a result.
        :param count: The number of tasks required to execute.
        """
        self.manager = mp.Manager()
        self.shared_dict: DictProxy[I, O] = self.manager.dict()
        self.condition = self.manager.Condition()
        self.pool = mp.Pool()
        self.inputs = inputs
        self.count = 0

        def done(result: tuple[I, O]):
            input, output = result
            try:
                with self.condition:
                    self.shared_dict[input] = output
                    self.condition.notify_all()
            except EOFError:
                # Closed semaphore
                pass

        for i in inputs:
            self.count += 1
            self.pool.apply_async(
                task,
                args=(i,),
                callback=done,
                error_callback=lambda e: print(f"Error in task {i}: {e}")
            )
        self.pool.close()

    def stream(self):
        """
        Generator that yields results as they become available.

        :return: A generator yielding lists of particles.
        """
        for input in self.inputs:
            try:
                with self.condition:
                    self.condition.wait_for(lambda: input in self.shared_dict)
                    result = self.shared_dict.pop(input)
                    yield result
            except FileNotFoundError:
                # If the pool is closed, we stop yielding results
                break

    def close(self):
        self.pool.terminate()
        self.manager.shutdown()
