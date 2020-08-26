document.getElementById("hamburger").addEventListener("click", () => {
    let x = document.getElementById("nav-items").className;
    document.getElementById("nav-items").className = x == "items-hidden" ? "items-visible" : "items-hidden";
});