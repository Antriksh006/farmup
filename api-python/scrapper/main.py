import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import logging
from datetime import date

# ---------------------------- CONFIG ----------------------------
CROPS = {
    "Tomato": ["https://www.commodityonline.com/mandiprices/tomato/odisha/"],
    "Paddy": ["https://www.commodityonline.com/mandiprices/paddy-dhan-common/odisha/"],
    "Brinjal": ["https://www.commodityonline.com/mandiprices/brinjal/odisha/"],
    "Potato": ["https://www.commodityonline.com/mandiprices/potato/odisha/"],
    "Onion": ["https://www.commodityonline.com/mandiprices/onion/odisha/"],
    "Maize": ["https://www.commodityonline.com/mandiprices/maize/odisha/"],
    "Wheat": ["https://www.commodityonline.com/mandiprices/wheat/odisha/"],
    "Chilli": ["https://www.commodityonline.com/mandiprices/chilli/odisha/"],
    "Cabbage": ["https://www.commodityonline.com/mandiprices/cabbage/odisha/"],
    "Cauliflower": ["https://www.commodityonline.com/mandiprices/cauliflower/odisha/"],
    "Carrot": ["https://www.commodityonline.com/mandiprices/carrot/odisha/"],
    "Banana": ["https://www.commodityonline.com/mandiprices/banana/odisha/"],
    "Sugarcane": ["https://www.commodityonline.com/mandiprices/sugarcane/odisha/"],
    "Groundnut": ["https://www.commodityonline.com/mandiprices/groundnut/odisha/"],
    "Mustard": ["https://www.commodityonline.com/mandiprices/mustard/odisha/"],
}

DATA_DIR = "data"
LOG_DIR = "logs"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 1  # seconds

# ---------------------------- LOGGING ----------------------------
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "scraper.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------- SCRAPER ----------------------------
scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

def parse_html_to_df(html_text: str) -> pd.DataFrame | None:
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if not table:
        return None

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    if "Market" not in headers:
        headers.insert(0, "Market")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            market_name = cols[0] if len(cols) > 0 else "Unknown"
            rows.append([market_name] + cols[1:] if len(cols) > 1 else [market_name])

    if not rows:
        return None
    return pd.DataFrame(rows, columns=headers)

def fetch_crop_data(crop_name: str, urls: list) -> list:
    all_data = []
    for url in urls:
        logging.info(f"Fetching {crop_name} from {url}")
        try:
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = parse_html_to_df(resp.text)
            if df is not None:
                all_data.append(df)
                logging.info(f"Fetched {len(df)} rows for {crop_name}")
            else:
                logging.warning(f"No table found for {crop_name} at {url}")
        except Exception as e:
            logging.error(f"Failed to fetch {url}: {e}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return all_data

# ---------------------------- MAIN ----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for crop, urls in CROPS.items():
        logging.info(f"Starting scrape for {crop}")
        all_dfs = fetch_crop_data(crop, urls)
        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            filename = os.path.join(DATA_DIR, f"{crop.lower()}_odisha_{date.today()}.csv")
            full_df.to_csv(filename, index=False)
            logging.info(f"Saved data → {filename}")
        else:
            logging.warning(f"No data found for {crop}")

    logging.info("🎉 Scraping complete!")

if __name__ == "__main__":
    main()
