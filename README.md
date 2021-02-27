# Face2BMI-modelgen

Core to Generate Models and Training logic

## Setup

### Clone with submodule

```sh
git clone --recurse-submodules https://github.com/face2bmi-ZmFjZTJibWk/Face2BMI-modelgen
```

### Creating a virtual environment

On macOS and Linux:

`python3 -m venv env`

On Windows:

`py -m venv env`

### Activating a virtual environment

On macOS and Linux:

`source env/bin/activate`

On Windows:

`.\env\Scripts\activate`

### Leaving the virtual environment

`deactivate`

### Installing Dependencies

`pip install -r requirements.txt`

### Updating Submodule `data`

`git submodule update --remote --merge`

## Data Folder's Structure

```
.(data root)
├── images
│   ├── <slug0>
│   │   ├── <slug1>_001.jpeg
│   │   ├── <slug1>_002.jpeg
│   │   └── <slug1>_<n>.jpg
│   ├── <slug1>
│   │   ├── <slug1>_001.jpg
│   │   ├── <slug1>_002.jpg
│   │   └── <slug1>_<n>.jpg
│   └── <slugn>
│       ├── <slugn>_001.jpg
│       ├── <slugn>_002.jpg
│       └── <slugn>_<n>.jpg
├── images.csv
└── README.md
```

## images.csv structure as example

| Name     | slug  | num of Photo | Weight (kg) | Height (m) | Actual BMI  | Type       |
| -------- | ----- | ------------ | ----------- | ---------- | ----------- | ---------- |
| Name one | slug0 | 5            | 90          | 1.68       | 31.8877551  | Obese      |
| Name two | slug1 | 9            | 82          | 1.77       | 26.17383255 | Overweight |
| Name n   | slugn | 7            | 55          | 1.6        | 21.484375   | Normal     |
