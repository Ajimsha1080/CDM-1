# CDM-1 Automation Makefile
.PHONY: all lint test no-skips verify docker-smoke manifest clean

PYTHON ?= python

all: lint test no-skips verify

manifest:
	@echo "==> Regenerating and Signing Release Manifest..."
	$(PYTHON) scripts/generate_and_sign_manifest.py

lint:
	@echo "==> Running Lint and Compliance Checks..."
	pip-licenses --from=mixed --fail-on="GPL;AGPL;LGPL;SSPL"

test:
	@echo "==> Running Full Test Suite with JUnit XML Generation..."
	pytest -rs --strict-markers -m "not requires_docker and not requires_weights" --junitxml=test-results.xml

no-skips:
	@echo "==> Enforcing Zero-Skips Verification Gate..."
	$(PYTHON) scripts/check_no_skips.py test-results.xml

verify:
	@echo "==> Verifying Release Manifest Integrity..."
	$(PYTHON) scripts/check_manifest_integrity.py
	@echo "==> Running End-to-End Smoke Test..."
	$(PYTHON) scripts/smoke_test.py

docker-smoke:
	@echo "==> Running Containerized Docker Smoke Test..."
	docker compose up -d --build
	$(PYTHON) scripts/smoke_test.py || (docker compose down && exit 1)
	docker compose down

clean:
	@echo "==> Cleaning Build Artifacts..."
	rm -rf test-results.xml docker-results.xml weights-results.xml .pytest_cache htmlcov .coverage dist build *.egg-info
