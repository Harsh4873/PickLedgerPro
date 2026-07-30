type Theme = 'light' | 'dark';

const root = document.documentElement;
const themeMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
const themeButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-theme-option]'));
const systemTheme = window.matchMedia('(prefers-color-scheme: light)');
const rail = document.querySelector<HTMLElement>('.app-rail');
const menuButton = document.querySelector<HTMLButtonElement>('[data-menu-toggle]');
const closeButton = document.querySelector<HTMLButtonElement>('[data-menu-close]');

function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem('harsh-theme');
    return value === 'light' || value === 'dark' ? value : null;
  } catch {
    return null;
  }
}

function applyTheme(theme: Theme, persist = false): void {
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  themeMeta?.setAttribute('content', theme === 'light' ? '#f3f1ec' : '#151513');

  themeButtons.forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.themeOption === theme));
  });

  if (persist) {
    try {
      window.localStorage.setItem('harsh-theme', theme);
    } catch {
      // The selected theme still applies for the current visit.
    }
  }
}

const initialTheme = root.dataset.theme === 'light' || root.dataset.theme === 'dark'
  ? root.dataset.theme
  : storedTheme() ?? (systemTheme.matches ? 'light' : 'dark');

applyTheme(initialTheme);

themeButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const requestedTheme = button.dataset.themeOption;
    if (requestedTheme === 'light' || requestedTheme === 'dark') {
      applyTheme(requestedTheme, true);
    }
  });
});

systemTheme.addEventListener('change', (event) => {
  if (!storedTheme()) applyTheme(event.matches ? 'light' : 'dark');
});

function setMenu(open: boolean): void {
  rail?.setAttribute('data-mobile-open', String(open));
  menuButton?.setAttribute('aria-expanded', String(open));
  if (menuButton) menuButton.textContent = open ? 'Close' : 'Menu';
  document.body.classList.toggle('mobile-menu-open', open);
}

menuButton?.addEventListener('click', () => setMenu(rail?.dataset.mobileOpen !== 'true'));
closeButton?.addEventListener('click', () => setMenu(false));
document.querySelectorAll<HTMLAnchorElement>('.app-nav-link').forEach((link) => {
  link.addEventListener('click', () => setMenu(false));
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') setMenu(false);
});

const year = document.querySelector<HTMLElement>('#current-year');
if (year) year.textContent = String(new Date().getFullYear());
