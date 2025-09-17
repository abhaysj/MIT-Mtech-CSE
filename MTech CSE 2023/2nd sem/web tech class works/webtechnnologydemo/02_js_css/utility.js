function validate() {
    var txtName = document.getElementById("txtName");
    if (txtName.value === "") {
        p.innerText("name cannotbe blank")
        alert("Please enter a name.");
        return false;
    }
    return true;
}  