// Landing page behaviour: scroll reveals, the word-by-word tagline, and the mobile menu. No scroll listeners: every
// reveal runs off IntersectionObserver.
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

// fade-up reveals, once per element
const rv = new IntersectionObserver(entries => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); rv.unobserve(e.target); }
}, { rootMargin: "0px 0px -5% 0px" });
document.querySelectorAll(".rv").forEach(el => reduce ? el.classList.add("in") : rv.observe(el));

// tagline: each word lights up as it crosses a line 40% up from the bottom of the screen, in reading order
const tag = document.querySelector(".reveal-words");
if (tag) {
  tag.innerHTML = tag.textContent.trim().split(/\s+/).map(w => `<span class="w">${w}</span>`).join(" ");
  const words = [...tag.querySelectorAll(".w")];
  if (reduce) words.forEach(w => w.classList.add("on"));
  else {
    const io = new IntersectionObserver(entries => {
      const hits = entries.filter(e => e.isIntersecting).map(e => e.target).sort((a, b) => words.indexOf(a) - words.indexOf(b));
      hits.forEach((w, i) => { w.style.transitionDelay = `${i * 60}ms`; w.classList.add("on"); io.unobserve(w); });
    }, { rootMargin: "0px 0px -40% 0px" });
    words.forEach(w => io.observe(w));
  }
}

// mobile menu: the burger's two lines turn into an X; links rise in one after another
const burger = document.querySelector(".burger"), menu = document.getElementById("menu");
function setMenu(open) {
  burger.setAttribute("aria-expanded", open);
  burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  document.body.style.overflow = open ? "hidden" : "";
  if (open) { menu.hidden = false; requestAnimationFrame(() => requestAnimationFrame(() => menu.classList.add("open"))); }
  else { menu.classList.remove("open"); setTimeout(() => { if (!menu.classList.contains("open")) menu.hidden = true; }, reduce ? 0 : 700); }
}
burger.addEventListener("click", () => setMenu(burger.getAttribute("aria-expanded") !== "true"));
menu.addEventListener("click", e => { if (e.target.closest("a")) setMenu(false); });
addEventListener("keydown", e => { if (e.key === "Escape" && burger.getAttribute("aria-expanded") === "true") setMenu(false); });
