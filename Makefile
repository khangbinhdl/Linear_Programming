.PHONY: make-install make-run

PORT ?= 8501

make-install:
	pip install -r requirements.txt

make-run:
	streamlit run frontend/app.py --server.port $(PORT)
