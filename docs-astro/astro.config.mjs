// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

/**
 * GitHub Pages (project site): set SITE_URL and BASE_PATH in CI, e.g.
 * SITE_URL=https://<owner>.github.io  BASE_PATH=/<repository>
 * Local dev: omit both; base defaults to "/".
 */
const site = process.env.SITE_URL;
const base = process.env.BASE_PATH || '/';

// https://astro.build/config
export default defineConfig({
	...(site ? { site } : {}),
	base,
	integrations: [
		starlight({
			title: 'Faraday',
			customCss: ['./src/styles/custom.css'],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/AscentBio/faraday-oss' },
			],
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'Welcome', slug: '' },
						{ label: 'Install', slug: 'install' },
						{ label: 'Quickstart', slug: 'getting-started' },
					],
				},
				{
					label: 'Run Tasks',
					autogenerate: { directory: 'run-queries' },
				},
				{
					label: 'Tools',
					autogenerate: { directory: 'tools' },
				},
				{
					label: 'Deploy',
					autogenerate: { directory: 'integrations' },
				},
				{
					label: 'Use cases',
					autogenerate: { directory: 'use-cases' },
				},
				{
					label: 'Examples',
					autogenerate: { directory: 'examples' },
				},
				{
					label: 'Reference',
					items: [
						{ label: 'CLI reference', slug: 'reference/cli' },
						{ label: 'Environment variables', slug: 'reference/environment-variables' },
						{ label: 'Runtime config', slug: 'reference/runtime-config' },
						{ label: 'YAML guidance', slug: 'reference/yaml-guidance' },
						{
							label: 'Python SDK',
							items: [
								{ label: 'Overview', slug: 'sdk' },
								{ label: 'First query', slug: 'sdk/first-query' },
								{ label: 'Follow-up conversations', slug: 'sdk/follow-up-conversations' },
							],
						},
					],
				},
			],
		}),
	],
});
