document.addEventListener("DOMContentLoaded", function() {
    const professionalDetails = document.querySelector(".professionalDetails");
    const professionalDetailsHeading = document.querySelector(".professionalDetails .heading");

    professionalDetailsHeading.addEventListener("click", function() {
        professionalDetails.classList.toggle("hidden");
    });
});
