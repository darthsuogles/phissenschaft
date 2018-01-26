
fn main() {
    let mut v = Vec::new();
    v.push("Hello, ");
    //let x = &v[0];
    v.push(" world!");
    println!("{}", v.join("=>"));
}
