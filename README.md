# useful-code-snippets
A searchable collection of useful little pieces of code

I'm saving any sort of useful fragment of code in this repo.

You can do a basic keyword search across all of the scripts using [search.py](./search.py), for example:

```bash
$ python search.py "google cloud bigquery to cloud storage"
```

Your query is compared against the script tags (tags are at the top of each code snippet script).

Words joined by underscore in the search query are considered a single phrase e.g. google_cloud_storage is interpreted by the search as "google cloud storage"

Current code snippets available:

```bash
snippets
├── bash
│   └── gcp
│       └── setup_python_on_ec2_virtual_machine.sh
└── python
    ├── gcp
    │   ├── all_bigquery_tables_to_cloud_storage_jsonl.py
    │   ├── bigquery_to_local_jsonl.py
    │   └── upload_file_to_cloud_storage_bucket.py
    ├── graph
    │   └── find_connected_node_paths.py
    └── misc
        └── dict_pretty_print.py
```
