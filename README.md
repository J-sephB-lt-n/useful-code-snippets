# useful-code-snippets

A searchable collection of useful little pieces of code

I'm saving any sort of useful fragment of code in this repo.

You can do a basic keyword search across all of the scripts using [search.py](./search.py), for example:

```bash
python search.py "google cloud bigquery to cloud storage"
```

Your query is compared against the script tags (tags are at the top of each code snippet script).

Words joined by underscore in the search query are considered a single phrase e.g. google_cloud_storage is interpreted by the search as "google cloud storage"

Current code snippets available:

```bash
── snippets
│   ├── bash
│   │   ├── gcp
│   │   │   └── setup_python_on_ec2_virtual_machine.sh
│   │   ├── llm
│   │   │   └── host_local_models_with_llama_cpp.sh
│   │   └── ubuntu
│   │       └── install_chrome_browser_on_ubuntu.sh
│   ├── docker
│   │   └── python
│   │       └── install_chrome_browser_in_python_docker_container
│   └── python
│       ├── dashboard
│       │   ├── dash
│       │   │   ├── dash_basic_auth.py
│       │   │   ├── dash_element_tiling.py
│       │   │   └── dash_multi_tab_app.py
│       │   └── streamlit
│       │       ├── streamlit_layout_tiling_example.py
│       │       └── streamlit_on_gcp_cloud_run.md
│       ├── data
│       │   └── native_groupby_agg.py
│       ├── email
│       │   └── extract_info_from_mime_string.py
│       ├── gcp
│       │   ├── all_bigquery_tables_to_cloud_storage_jsonl.py
│       │   ├── bigquery_to_local_jsonl.py
│       │   ├── self_deleting_vm.py
│       │   ├── upload_file_to_cloud_storage_bucket.py
│       │   └── write_to_gcp_cloud_storage_from_outside_gcp.py
│       ├── graph
│       │   └── find_connected_node_paths.py
│       ├── html
│       │   ├── download_webpage_content_javascript_render.py
│       ├── llm
│       │   └── count_tokens_mistral_model.py
│       ├── misc
│       │   ├── code_section_timer.py
│       │   └── dict_pretty_print.py
│       ├── nlp
│       │   └── text_to_bag_of_words_nltk.py
│       └── spark
│           └── html_word_counter_gcp_dataproc_gcp_cloud_storage_bs4_pyspark.md
```
