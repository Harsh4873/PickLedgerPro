const rail = document.querySelector<HTMLElement>('.app-rail');
const menuButton = document.querySelector<HTMLButtonElement>('[data-menu-toggle]');
const closeButton = document.querySelector<HTMLButtonElement>('[data-menu-close]');

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
