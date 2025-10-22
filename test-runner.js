const puppeteer = require('puppeteer');

async function runTest() {
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.goto('http://localhost:8000/test-fix.html');

    await page.waitForSelector('.test-section');
    const testResult = await page.evaluate(() => {
        return document.querySelector('#testResult').textContent;
    });

    console.log(testResult);

    if (testResult.includes('Failed')) {
        process.exit(1);
    }

    await browser.close();
}

runTest();
