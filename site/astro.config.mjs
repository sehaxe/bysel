import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	site: 'https://sehaxe.github.io',
	base: '/busel-ai/',

	integrations: [
		starlight({
			title: 'Busel AI',
			description:
				'Sovereign 1.58-bit LLM with mAR residuals, hybrid GDN-2/MLA attention, byte-level patching, and MTP-4. Trains on consumer hardware.',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/sehaxe/busel-ai' },
			],
			sidebar: [
				{
					label: 'Get Started',
					items: [
						{ slug: 'guides/getting-started', label: 'Installation & quick start' },
						{ slug: 'guides/quick-tour', label: 'A 5-minute tour' },
						{ slug: 'guides/profiles', label: 'Choosing a profile' },
					],
				},
				{
					label: 'Architecture',
					items: [
						{ slug: 'architecture/overview', label: 'Overview & design philosophy' },
						{ slug: 'architecture/one-bit-weights', label: '1.58-bit weights (BitLinear + H_BitLinear)' },
						{ slug: 'architecture/patching', label: 'Byte-level patching (FastBLT)' },
						{ slug: 'architecture/attention', label: 'Hybrid attention (GDN-2 + MLA)' },
						{ slug: 'architecture/mar', label: 'mAR — Manifold Constrained Attention Residuals' },
						{ slug: 'architecture/moe', label: 'MoE with Blackboard Memory' },
						{ slug: 'architecture/mtp', label: 'Multi-Token Prediction (MTP-4)' },
					],
				},
				{
					label: 'Training',
					items: [
						{ slug: 'training/training-guide', label: 'How a training run works' },
						{ slug: 'training/optimizer', label: 'Hybrid Muon + AdamW' },
						{ slug: 'training/autopilot', label: 'buselAutoPilot v6.0' },
						{ slug: 'training/curriculum', label: 'Curriculum & Chinchilla' },
						{ slug: 'training/checkpointing', label: 'Checkpointing & resume' },
					],
				},
				{
					label: 'Data',
					items: [
						{ slug: 'data/pipeline', label: 'Stream-interleaving pipeline' },
						{ slug: 'data/formats', label: 'Supported file formats' },
						{ slug: 'data/multimodal', label: 'Multimodal encoding' },
					],
				},
				{
					label: 'API Reference',
					items: [
						{ slug: 'reference/model', label: 'Model classes' },
						{ slug: 'reference/training', label: 'Training components' },
						{ slug: 'reference/data', label: 'Data pipeline' },
						{ slug: 'reference/registry', label: 'Plug-in registry' },
						{ slug: 'reference/logging', label: 'Structured event log' },
						{ slug: 'reference/ui', label: 'UI / Teto helpers' },
						{ slug: 'reference/config', label: 'Config profiles' },
					],
				},
				{
					label: 'Performance',
					items: [
						{ slug: 'performance/compile-modes', label: 'torch.compile modes' },
						{ slug: 'performance/hardware', label: 'Hardware tuning' },
						{ slug: 'performance/profiling', label: 'Profiling a step' },
					],
				},
				{
					label: 'Operations',
					items: [
						{ slug: 'operations/inference', label: 'Running inference' },
						{ slug: 'operations/troubleshooting', label: 'Troubleshooting' },
						{ slug: 'operations/faq', label: 'FAQ' },
					],
				},
			],
			customCss: [],
		}),
	],
});
