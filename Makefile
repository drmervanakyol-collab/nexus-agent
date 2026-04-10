.PHONY: lint test check ci golden bench property mutmut

lint:
	ruff check nexus/ && mypy nexus/

test:
	pytest tests/unit/

check: lint test

ci: lint test
	pytest tests/integration/
	bandit -r nexus/ -q
	safety check

golden:
	NEXUS_GOLDEN_TESTS=1 pytest tests/golden/ -v -s

bench:
	pytest tests/benchmarks/

property:
	pytest tests/property/ -v

mutmut:
	python scripts/run_mutation.py --paths nexus/core/ --tests tests/property/ tests/integration/ --threshold 0.70
