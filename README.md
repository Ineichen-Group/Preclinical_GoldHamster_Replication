# Preclinical_goldhamster

## Goldhamster-DS Model training

We wanted to replicate the study by [Neves M, Klippert A, Knöspel F, et al. ](https://pubmed.ncbi.nlm.nih.gov/37658458/), following the steps:

1. Download the annotated data splits from [zenodo](https://doi.org/10.5281/zenodo.7152295) in [./data/corpus_annotations](./data/corpus_annotations/).
2. Re-implement the cross-validation training code from [github](https://github.com/mariananeves/goldhamster) in order to use newer python and PyTorch versions, because we the original code did not recognize the GPU ressources.
   - original code: [./scripts/train_bert_abstract_classifier_goldhamster.py](./scripts/train_bert_abstract_classifier_goldhamster.py)
   - adapted code: [./scripts/train_bert_abstract_classifier_goldhamster_pytorch.py](./scripts/train_bert_abstract_classifier_goldhamster_pytorch.py)
3. Train three BERT domain-specific models, e.g, see [./scripts/server/run_single_goldhamster_biobert.sh](./scripts/server/run_single_goldhamster_biobert.sh).
4. Compare the model performance, see [./Goldhamster_eval_models.ipynb](./Goldhamster_eval_models.ipynb)

We chose the BioLinkBERT model due to it's slightly better performance.


## Collecting full data from PubMed 

We used the EDirect package, which includes several commands that use the E-utilities API to find and retrieve PubMed data. You can install it via the command:
```
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
```

Note:  For best performance, obtain an API Key from NCBI, and place the following line in your .bash_profile and .zshrc configuration files (follow https://support.nlm.nih.gov/kbArticle/?pn=KA-05317):
```
  export NCBI_API_KEY=unique_api_key
```
1. To obtain the initial set of relevant PMIDs, the database was queried using several search string related to CNS and Psychiatric conditions, as follows:
- CNS related free text search queries found in [./data/pubmed_queries/nervous_system/](./data/pubmed_queries/nervous_system/). Those were executed iteratively with the script [./scripts/pubmed_query_long.sh](./scripts/pubmed_query_long.sh). In addition MeSH based queries are in [./data/pubmed_queries/nervous_system/cns_mesh_1.txt](./data/pubmed_queries/nervous_system/cns_mesh_1.txt) and [./data/pubmed_queries/nervous_system/cns_mesh_2.txt](./data/pubmed_queries/nervous_system/cns_mesh_2.txt)

- Psychiatric conditions related queries: see [./data/pubmed_queries/psychiatric/psychiatrich_query_used.txt](./data/pubmed_queries/psychiatric/psychiatrich_query_used.txt).


The queries use the following syntax:
```
esearch -db pubmed -query '(Central nervous system diseases[MeSH] OR Mental Disorders OR Psychiatric illness[MeSH]) AND English[lang]' | efetch -format uid > "./cns_psychiatric_diseases_pmids_en_$(date +%Y%m%d).txt"
```

The outputs from those queries were stored in [./data/pubmed_queries/results_pmids/](./data/pubmed_queries/results_pmids/). They were then combined into a full list of PMIDs in [./PubMed_Query_PMIDs_List.ipynb](./PubMed_Query_PMIDs_List.ipynb).


2. Split into chunks of 5000 PMIDs per file, see [./scripts/server/split_pmids_to_chunks.sh](./scripts/split_pmids_to_chunks.sh)

3. Run parallel fetching of content on the surver, see [./scripts/server/fetch_pubmed_data_large.sh](./scripts/fetch_pubmed_data_large.sh
)

This executes the following call:
```
efetch -db pubmed -id "$id_list" -format xml | \
    xtract -pattern PubmedArticle -tab '^!^' -def "N/A" \
    -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText \
    -block ArticleId -if ArticleId@IdType -equals doi -element ArticleId \
    > "$OUTPUT_FILE"
```

Relevant API documentation references:

- https://www.ncbi.nlm.nih.gov/books/NBK179288/
- https://www.nlm.nih.gov/dataguide/edirect/xtract_formatting.html
- https://dataguide.nlm.nih.gov/classes/edirect-for-pubmed/samplecode4.html#specify-a-placeholder-to-replace-blank-spaces-in-the-output-table