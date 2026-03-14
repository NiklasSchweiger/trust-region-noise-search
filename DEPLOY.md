# Publishing the project page

The page will be live at: **https://niklasschweiger.github.io/trust-region-noise-search/**

## Option A: Use a `gh-pages` branch (recommended)

Keeps the website separate from your source code on `main`.

1. **From this folder** (`trust-region-noise-search` with the website files), ensure your remote points to your repo:
   ```bash
   git remote -v
   ```
   If it shows `niklasschweiger/trust-region-noise-search`, you're set. If not:
   ```bash
   git remote set-url origin https://github.com/niklasschweiger/trust-region-noise-search.git
   ```

2. **Create and push the `gh-pages` branch** (website only):
   ```bash
   git checkout -b gh-pages
   git add index.html static .nojekyll
   git commit -m "Add project page"
   git push -u origin gh-pages
   ```

3. **Enable GitHub Pages**  
   On GitHub: **Settings → Pages → Source**: choose **Deploy from a branch**, branch **gh-pages**, folder **/ (root)** → Save.

4. After a minute or two, the site will be at https://niklasschweiger.github.io/trust-region-noise-search/

---

## Option B: Use the `docs/` folder on `main`

If you prefer the site to live next to your code in the same branch:

1. In your **source code** repo, create a `docs` folder.
2. Copy into `docs/`: `index.html`, the whole `static/` folder, and `.nojekyll`.
3. Commit and push to `main`.
4. On GitHub: **Settings → Pages → Source**: choose **Deploy from a branch**, branch **main**, folder **/docs** → Save.

---

**Note:** Add your images under `static/images/` (e.g. `Teaser_Figure.png`, `image_results.png`, `Teaser_image.png`) so they load correctly.
