import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	// 🎯 ДОБАВЛЕНЫ ДВА КРИТИЧЕСКИХ ПАРАМЕТРА ДЛЯ ДЕПЛОЯ НА GITHUB PAGES:
	site: 'https://sehaxe.github.io', // Ваш домен на GitHub Pages
	base: '/busel-ai/',               // Имя репозитория (косые черты в начале и конце обязательны!)
	
	integrations: [
		starlight({
			title: 'Busel AI',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/sehaxe/busel-ai' }
			],
			sidebar: [
				{
					label: 'Guides',
					items: [
						'guides/getting-started',
					],
				},
			],
		}),
	],
});