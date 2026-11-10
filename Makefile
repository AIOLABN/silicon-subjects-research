PY ?= .venv/bin/python
SIM ?= .venv/bin/sim
CONFIG ?= configs/study.yaml

.PHONY: install lock lint typecheck test check demo verify

install:
	python3 -m venv .venv && .venv/bin/pip install --require-hashes -r requirements.lock && \
	  .venv/bin/pip install --no-deps --no-build-isolation -e .

lock:
	printf "hatchling>=1.25\neditables>=0.5\n" | uv pip compile pyproject.toml - --extra anthropic --extra dev \
	  --extra sources --universal --python-version 3.11 --generate-hashes --no-header \
	  -o requirements.lock

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

typecheck:
	.venv/bin/mypy src

test:
	.venv/bin/pytest -q -n auto

check: lint typecheck test

demo:
	$(SIM) verify -c configs/demo.yaml || $(SIM) freeze -c configs/demo.yaml --force
	rm -rf results/demo prereg.demo-mock.runs.jsonl
	$(SIM) run -c configs/demo.yaml --run-id demo
	$(SIM) analyze demo -c configs/demo.yaml
	@echo "open results/demo/report.md"

verify:
	$(MAKE) check
	-$(SIM) sources fetch
	$(SIM) sources verify
	@tmp=$$(mktemp -d) && sim=$(abspath $(SIM)) && \
	  tar --exclude='./data/raw' --exclude='./data/survey/raw' -cf - ./configs ./data | tar -C $$tmp -xf - && \
	  cd $$tmp && $$sim freeze -c configs/demo.yaml --force >/dev/null && \
	  $$sim run -c configs/demo.yaml --run-id verify >/dev/null && \
	  $$sim analyze verify -c configs/demo.yaml >/dev/null && \
	  test -s results/verify/report.md && echo "demo pipeline: OK (isolated in $$tmp)" && rm -rf $$tmp
	-$(SIM) doctor -c $(CONFIG)
