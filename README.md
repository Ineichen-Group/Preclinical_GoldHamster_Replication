# Preclinical_goldhamster

## Collecting full data from PubMed 

We used the EDirect package, which includes several commands that use the E-utilities API to find and retrieve PubMed data. You can install it via the command:
```
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
```

Note:  For best performance, obtain an API Key from NCBI, and place the following line in your .bash_profile and .zshrc configuration files (follow https://support.nlm.nih.gov/kbArticle/?pn=KA-05317):
```
  export NCBI_API_KEY=unique_api_key
```
1. To obtain the initial set of relevant PMIDs, the database was queried using a generic search string related to CNS and Psychiatric conditions, as follows:

```
esearch -db pubmed -query '(Central nervous system diseases[MeSH] OR Mental Disorders Psychiatric illness[MeSH]) AND English[lang]' | efetch -format uid > ./pubmed/cns_psychiatric_diseases_pmids_en.txt
```

2. Split into chunks of 5000 PMIDs per file, see [./scripts/split_pmids_to_chunks.sh](./scripts/split_pmids_to_chunks.sh)

3. Run parallel fetching of content on the surver, see [./scripts/fetch_pubmed_data_large.sh](./scripts/fetch_pubmed_data_large.sh
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