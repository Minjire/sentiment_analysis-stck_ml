# this script collects news headlines for given companies. source of data is the kenyanwallstreet website

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd

# open chrome and navigate to kenyan wallstreet website
ChromeDriver = webdriver.Chrome()
ChromeDriver.get("https://kenyanwallstreet.com/#")
ChromeDriver.maximize_window()

wait = WebDriverWait(ChromeDriver, 5)
presence = EC.presence_of_element_located
visible = EC.visibility_of_element_located
wait.until(visible((By.TAG_NAME, "iframe")))
ChromeDriver.switch_to.frame(ChromeDriver.find_element_by_tag_name("iframe"))
ChromeDriver.refresh()
companies = ['safaricom', 'equity', 'eabl', 'kcb']

for company in companies:
    try:
        wait.until(visible((By.NAME, "s")))
        ChromeDriver.find_element_by_name('s').send_keys(company)
        wait.until(visible((By.CLASS_NAME, "jeg_search_button")))
        ChromeDriver.find_element_by_class_name("jeg_search_button").click()
        data = []
        validator = True

        while validator is True:
            wait.until(visible((By.CLASS_NAME, "jnews_search_content_wrapper")))
            container = ChromeDriver.find_element_by_class_name('jeg_posts')
            articles = container.find_elements_by_tag_name('article')

            for article in articles:
                header = article.find_element_by_class_name('jeg_post_title')
                date = article.find_element_by_class_name('jeg_meta_date')
                data.append([header.text, date.text])

            try:
                ChromeDriver.find_element_by_class_name('next').click()
            except NoSuchElementException:
                validator = False

        df = pd.DataFrame(data, columns=['News', 'Date'])
        df.to_csv(company + '.csv', index=False)
        ChromeDriver.find_element_by_name('s').clear()
    finally:
        print('done')
