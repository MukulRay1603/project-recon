// Screenshots of the SAMPLE-DATA UI (scripts/ui_sample_app.py must be running).
// node scripts/ui_screenshots.js [url] [outdir]
const { chromium } = require('playwright');
(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:7862';
  const out = process.argv[3] || 'docs/ui';
  const b = await chromium.launch();
  for (const [name, w, h] of [['desktop', 1280, 900], ['phone', 390, 844]]) {
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await p.goto(url, { waitUntil: 'load', timeout: 60000 });
    await p.waitForSelector('textarea', { timeout: 30000 });
    await p.waitForTimeout(1500);
    await p.screenshot({ path: `${out}/${name}-empty.png`, fullPage: true });
    await p.locator('textarea').first().fill('What is the current state of KV cache compression for long-context LLMs?');
    await p.getByRole('button', { name: 'Research', exact: true }).click();
    await p.getByText('Answer written').waitFor({ timeout: 30000 });
    await p.waitForTimeout(1500);
    await p.screenshot({ path: `${out}/${name}-result.png`, fullPage: true });
    await p.getByRole('tab', { name: 'Counter-evidence' }).click();
    await p.waitForTimeout(500);
    await p.screenshot({ path: `${out}/${name}-counter-evidence.png`, fullPage: true });
    const sw = await p.evaluate(() => document.documentElement.scrollWidth);
    console.log(name, 'viewport', w, 'scrollWidth', sw);
    await p.close();
  }
  await b.close();
})();
