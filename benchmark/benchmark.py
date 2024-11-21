import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Tuple
import numpy as np


logger = logging.getLogger(__name__)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

API_URL = 'http://127.0.0.1:8000/v1/completions'
MODEL_UID = 'Qwen2_7B'

HEADERS = {
    'Content-Type': 'application/json',
}


async def send_request(session, payload, prompt_len):
    request_start_time = time.time()
    async with session.post(API_URL, data=payload, headers=HEADERS) as response:
        if response.status == 200:
            result = await response.json()
            completion_tokens = len(result['choices'][0]['text'])
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_len, completion_tokens, request_latency))
            return result
        else:
            return {'error': response.status, 'message': await response.text()}


class BenchMarkRunner:

    def __init__(
        self,
        requests: List[Tuple[str, int, int]],  # prompt, prompt_len, completion_len
        concurrency: int,
    ):
        self.concurrency = concurrency
        self.requests = requests
        self.request_left = len(requests)
        self.request_queue = asyncio.Queue(concurrency or 100)

    async def run(self):
        tasks = []
        for i in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))
        for req in self.requests:
            await self.request_queue.put(req)
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    async def worker(self):
        timeout = aiohttp.ClientTimeout(total=5 * 60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self.request_left > 0:
                prompt = await self.request_queue.get()
                payload = json.dumps({"model": MODEL_UID,
                                      "prompt": f"Human: {prompt}\nAssistant: ",
                                      "stop": "<|endoftext|>",
                                      "temperature": 0,
                                      "max_tokens": 64,
                                      "top_k": 1
                                      })

                response = await send_request(session, payload, len(prompt))
                self.request_left -= 1
                print(f"Response {len(self.requests) - self.request_left}")


def main():
    concurrency = 50 
    logger.info("Preparing for benchmark.")
    testset = json.load(open("bench_data.json"))
    input_requests = list(testset.values()) 

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.4f} s")
    print(f"Throughput: {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.4f} s")
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.4f} s")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.4f} s")
    throughput = (
            sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    )
    print(f"Throughput: {throughput} tokens/s")


if __name__ == '__main__':
    main()
