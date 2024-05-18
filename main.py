import requests

from fastapi import FastAPI
from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    worker_urls: list[str] = ["http://worker1", "http://worker2",
                              "http://worker3"]
    coef_cpu: int = 1
    coef_gpu: int = 0
    coef_network: int = 0


settings = WorkerSettings()

app = FastAPI()

worker_state: list[dict | None] = [None for _ in
                                   range(len(settings.worker_urls))]


def argmax(it):
    _max = None
    max_idx = None
    first = True
    for idx, item in enumerate(it):
        if first:
            _max = item
            max_idx = idx
            first = False
            continue

        if item > _max:
            _max = item
            max_idx = idx

    return max_idx


def fetch_worker_state():
    for idx, url in enumerate(settings.worker_urls):
        resp = requests.get(url + "/server/load")
        body_json = resp.json()

        worker_state[idx] = {
            "cpu_perc": body_json["available_FLOPS_percentage"],
            "available_RAM": body_json["available_RAM"],
        }


# fetch_worker_state()


def pick_worker():
    k_storage = 1
    k_ram = 1
    scores = []
    for idx, w_state in enumerate(worker_state):
        score = (
                settings.coef_cpu * (w_state["cpu_perc"] / 100)
                + settings.coef_gpu * 0
                + settings.coef_network * 1
        )
        score = score * k_storage * k_ram
        scores.append(score)

    worker_idx = argmax(scores)
    return worker_idx


def run_task_in_worker(worker_idx):
    url = f"{settings.worker_urls[worker_idx]}/docker/run?image=cpu-bound&waited=true"
    resp = requests.post(url, json={"PRECISION": "55000"})
    body_json = resp.json()
    return body_json


@app.post("/api/v1/run")
def make_new_task():
    worker_idx = pick_worker()
    return run_task_in_worker(worker_idx)


@app.post("/api/v1/stop_all")
def stop_all_workers():
    for url in settings.worker_urls:
        url = f"{url}/docker/containers/all"
        resp = requests.delete(url)
        assert resp.status_code == 204

    return {"status": "ok"}
