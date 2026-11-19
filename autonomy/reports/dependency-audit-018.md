# Execution018 frontend dependency audit

Verified 2026-11-18 17:34 UTC with Node 24.15.0 / npm 11.12.1. Dependency versions and lockfile are unchanged. No advisory suppression, override, forced fix, downgrade, global setting or new dependency was added.

The initial execution018 `npm audit --json` probe before the interruption exited 1 with **4 high / 4 total**, all other severity counts zero. Its only underlying advisory was [GHSA-fc5q-39xc-vjpf / CVE-2026-112366](https://github.com/advisories/GHSA-fc5q-39xc-vjpf), for picomatch <=4.0.7. npm propagated that finding to fdir, tinyglobby and direct development dependency Vite; these were four affected package entries, not four independent advisories. npm suggested Vite 6.2.7, a downgrade from the existing 6.4.5, rather than a patched picomatch release.

On November 18 the [GitHub advisory](https://github.com/advisories/GHSA-fc5q-39xc-vjpf) was withdrawn and still lists no patched versions. The [upstream maintainer discussion](https://github.com/micromatch/picomatch/pull/235) disputes the reported vulnerability and the proposed fix PR was closed unmerged. Official [`npm view picomatch versions --json`](https://registry.npmjs.org/picomatch) confirmed 4.0.7 remains the latest published version. A fresh npm audit then exited 0 with **0 high / 0 total**, all severity counts zero. This is an upstream advisory withdrawal, not a locally applied security patch or a claim that arbitrary glob patterns are safe.

The installed tree remains Vite **6.4.5**, picomatch **4.0.7**, fdir **6.5.0**, tinyglobby **0.2.17**. Vite's picomatch range is ^4.0.2, tinyglobby's ^4.0.4 and fdir's optional peer ^3 || ^4. No published compatible patched version exists to pin. Keeping the existing pins is the smallest supported result now that the registry has removed the finding. React / react-dom remain **19.1.0**.

Validation after the registry withdrawal:

- `npm ci`: exit 0, clean installation, zero reported vulnerabilities.
- `npm test`: exit 0, all 9 tests passed.
- `npm run build`: exit 0; TypeScript and Vite production build passed, 33 modules transformed.
- Final `npm audit --json`: exit 0; vulnerabilities `{}`, info/low/moderate/high/critical/total all 0; metadata counts prod 4, dev 115, optional 53, total 118.
- `npm ls vite picomatch fdir tinyglobby --all`: exit 0, expected deduplicated versions above.
- Byte comparison before/after checks: package.json SHA256 `d5a07330aaeb542852a29e4d43c1cf4dca16d7db48125a8163f4c5c1505b043a`; package-lock.json SHA256 `a9c0a15c9de87e3af1f2c16e8c2338cb409d29f62cb87ae8cdd2d1887eb7f0ae`; both unchanged.

Only this report was authored for this increment. No UI, Python/ML source, experiment config or uv lock was edited. No model, data, accuracy, timestamp or advice gate changed. These automated checks are software verification, not user testing. Independent review remains the coordinator's responsibility.
