# useful-code-snippets

A searchable collection of useful little pieces of code

I'm saving any sort of useful fragment of code in this repo.

You can do a basic keyword search across all of the scripts using [search.py](./search.py), for example:

```bash
python search.py "google cloud bigquery to cloud storage"
```

Your query is compared against the script tags (tags are at the top of each code snippet script).

Words joined by underscore in the search query are considered a single phrase e.g. google_cloud_storage is interpreted by the search as "google cloud storage"

Available snippets:

```bash
├── R
│   └── dataviz
│       └── feature_distributions_by_continuous_target_variable.R
├── bash
│   ├── gcp
│   │   ├── list_n_random_filenames_from_cloud_storage_bucket.sh
│   │   └── setup_python_on_ec2_virtual_machine.sh
│   ├── llm
│   │   └── host_local_models_with_llama_cpp.sh
│   ├── misc
│   │   └── time_sections_in_bash_script.sh
│   └── ubuntu
│       └── install_chrome_browser_on_ubuntu.sh
├── css
│   └── tooltips.html
├── docker
│   ├── gcp
│   │   ├── flask_app_on_cloud_run.md
│   │   └── native_python_logging_in_compute_engine_vm_docker.py
│   ├── local_docker_builds.md
│   └── python
│       └── install_chrome_browser_in_python_docker_container
├── javascript
│   └── d3js
│       └── simple_line_chart.html
├── python
│   ├── altair
│   │   ├── plot_grid.py
│   │   └── scatterplot.py
│   ├── azure
│   │   └── blob_storage
│   │       ├── read_blob_contents_into_python_memory.py
│   │       └── write_python_obj_contents_to_blob.py
│   ├── cryptography
│   │   └── fernet_symmetric_encryption.py
│   ├── dashboard
│   │   ├── dash
│   │   │   ├── dash_basic_auth.py
│   │   │   ├── dash_element_tiling.py
│   │   │   └── dash_multi_tab_app.py
│   │   └── streamlit
│   │       ├── streamlit_layout_tiling_example.py
│   │       └── streamlit_on_gcp_cloud_run.md
│   ├── data
│   │   ├── native_groupby_agg.py
│   │   └── python_native_csv_writer.py
│   ├── data_structures_and_algs
│   │   └── len_of_longest_common_substring.py
│   ├── database
│   │   ├── basic_postgres_psycopg.py
│   │   └── basic_shelve_usage.py
│   ├── duckdb
│   │   └── gcp_bigquery_to_duckdb.md
│   ├── email
│   │   └── extract_info_from_mime_string.py
│   ├── gcp
│   │   ├── authorizing_rest_api_calls_to_gcp_resources.py
│   │   ├── bigquery
│   │   │   ├── all_bigquery_tables_to_cloud_storage_jsonl.py
│   │   │   ├── bigquery_to_local_jsonl.py
│   │   │   ├── bigquery_to_python.py
│   │   │   └── python_to_bigquery.py
│   │   ├── cloud_storage
│   │   │   ├── cloud_storage_to_python.py
│   │   │   ├── list_filepaths_in_cloud_storage_bucket.py
│   │   │   ├── python_object_to_cloud_storage.py
│   │   │   ├── upload_file_to_cloud_storage_bucket.py
│   │   │   └── write_to_gcp_cloud_storage_from_outside_gcp.py
│   │   └── self_deleting_vm.py
│   ├── graph
│   │   └── find_connected_node_paths.py
│   ├── gui
│   │   └── interactive_file_picker.py
│   ├── html
│   │   ├── download_webpage_content_javascript_render.py
│   │   ├── enumerate_html_tag_paths.py
│   │   ├── extract_css_from_webpage.py
│   │   └── extract_user_facing_text_from_html.py
│   ├── http
│   │   └── flask
│   │       ├── programmatically_host_local_flask_app.md
│   │       └── simple_auth_decorator.md
│   ├── hyperparameter_optimisation
│   │   └── optuna_hyperparam_tuning.py
│   ├── image
│   │   ├── arrange_images_in_grid.py
│   │   ├── extract_colour_palette_from_image.py
│   │   └── extract_colour_palette_from_webpage.py
│   ├── llm
│   │   ├── count_tokens_mistral_model.py
│   │   ├── llm_interface_class.py
│   │   ├── llm_retry_pattern.py
│   │   ├── openai_api
│   │   │   └── function_calling.py
│   │   └── streaming_llama_cpp_response.py
│   ├── misc
│   │   ├── code_section_timer.py
│   │   ├── create_wordbased_id.py
│   │   ├── dict_pretty_print.py
│   │   ├── dynamic_logger_format.py
│   │   ├── function_logging_decorator.py
│   │   └── function_retry_wrapper.py
│   ├── nlp
│   │   └── text_to_bag_of_words_nltk.py
│   ├── regex
│   │   └── extract_domain_from_website_url.py
│   ├── selenium
│   │   ├── ip_rotation_with_auth_proxy_server.py
│   │   └── selenium_on_gcp_dev_util.py
│   ├── sklearn
│   │   ├── pipeline_examples.py
│   │   └── save_and_load_models_joblib.py
│   ├── spark
│   │   └── html_word_counter_gcp_dataproc_gcp_cloud_storage_bs4_pyspark.md
│   ├── stats
│   │   └── find_best_fitting_univariate_distribution.py
│   └── templating
│       └── jinja2_basic_usage.py
├── terraform
│   └── gcp
│       └── create_gcp_service_account.md
└── vector_database
    └── qdrant
        ├── hybrid_search_python.py
        ├── python_qdrant_multitenancy_issue_example.py
        └── python_quickstart.md

50 directories, 76 files
```
