# Reddit User Persona Generator

This script analyzes a Reddit user's posts and comments to generate a detailed user persona using the Google Gemini API.

## Features

- Scrapes Reddit user's public posts and comments
- Uses Google Gemini to analyze demographics, interests, personality, and more
- Saves a clean persona report to a text file

## Requirements

- Python 3.8+
- Google Gemini API Key ([get one here](https://makersuite.google.com/app/apikey))

## Setup

### Clone the repo
```bash
git clone https://github.com/NayashaPrakash/Reddit-Persona_Generator.git
```
Move to project folder.

### Create virtual environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```
### Usage
```bash
python persona_generator.py <reddit_profile_url> --api-key <your_gemini_api_key>
```
Eg:
```bash
python persona_generator.py "https://www.reddit.com/user/Unidan" --api-key YOUR_API_KEY
```
