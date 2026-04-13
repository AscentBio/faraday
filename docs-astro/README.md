# Documentation (Astro)

[Starlight](https://starlight.astro.build/) static docs, deployable to [GitHub Pages](https://pages.github.com/) via `.github/workflows/deploy-docs.yml`.

```bash
npm install
npm run dev
```

Production build (optional: set `SITE_URL` and `BASE_PATH` as in the workflow):

```bash
SITE_URL=https://<owner>.github.io BASE_PATH=/<repo> npm run build
```

Enable **Settings → Pages → GitHub Actions** on the repository before the first deploy.
