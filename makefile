.PHONY: dvc-add

# List of folders to add to DVC
FOLDERS := ./checkpoints/embedding_models/ \
	./checkpoints/semantic_models/ \
	./data/terumo-data-testset/ \
	./data/terumo-data-training/ \
	./embeddings/


dvc-add:
	@for folder in $(FOLDERS); do \
		dvc add $$folder; \
	done

.PHONY: clean

clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;
