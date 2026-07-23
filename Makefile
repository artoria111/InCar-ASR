.PHONY: demo dashboard download-model test

demo:
	./scripts/run_demo.sh

download-model:
	python3 scripts/download_demo_model.py

dashboard:
	python3 apps/dashboard/server.py

test:
	./scripts/run_demo_tests.sh
