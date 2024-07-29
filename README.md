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
snippets
├── R
│   └── dataviz
│       └── feature_distributions_by_continuous_target_variable.R
├── bash
│   ├── gcp
│   │   ├── list_n_random_filenames_from_cloud_storage_bucket.sh
│   │   └── setup_python_on_ec2_virtual_machine.sh
│   ├── llm
│   │   ├── host_local_models_with_llama_cpp.sh
│   └── ubuntu
│       └── install_chrome_browser_on_ubuntu.sh
├── docker
│   └── python
│       └── install_chrome_browser_in_python_docker_container
└── python
    ├── dashboard
    │   ├── dash
    │   │   ├── dash_basic_auth.py
    │   │   ├── dash_element_tiling.py
    │   │   └── dash_multi_tab_app.py
    │   └── streamlit
    │       ├── streamlit_layout_tiling_example.py
    │       └── streamlit_on_gcp_cloud_run.md
    ├── data
    │   └── native_groupby_agg.py
    ├── email
    │   └── extract_info_from_mime_string.py
    ├── gcp
    │   ├── all_bigquery_tables_to_cloud_storage_jsonl.py
    │   ├── bigquery_to_local_jsonl.py
    │   ├── bigquery_to_python.py
    │   ├── cloud_storage_to_python.py
    │   ├── self_deleting_vm.py
    │   ├── upload_file_to_cloud_storage_bucket.py
    │   └── write_to_gcp_cloud_storage_from_outside_gcp.py
    ├── graph
    │   └── find_connected_node_paths.py
    ├── html
    │   ├── download_webpage_content_javascript_render.py
    │   ├── extract_css_from_webpage.py
    │   └── extract_user_facing_text_from_html.py
    ├── http
    │   └── flask
    │       └── programmatically_host_local_flask_app.md
    ├── image
    │   ├── extract_colour_palette_from_image.py
    │   ├── extract_colour_palette_from_webpage.py
    │   └── pillow
    │       └── arrange_images_in_grid.py
    ├── llm
    │   └── count_tokens_mistral_model.py
    ├── misc
    │   ├── code_section_timer.py
    │   ├── dict_pretty_print.py
    │   └── function_retry_wrapper.py
    ├── nlp
    │   └── text_to_bag_of_words_nltk.py
    ├── regex
    │   └── extract_domain_from_website_url.py
    ├── spark
    │   └── html_word_counter_gcp_dataproc_gcp_cloud_storage_bs4_pyspark.md
    └── stats
        └── find_best_fitting_univariate_distribution.py

29 directories, 38 files
```
