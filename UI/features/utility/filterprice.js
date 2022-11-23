function pricerange(range , actualprice) {   //returns true if actualprice is in range

    const price = actualprice.replace(/\D/g, ''); // 👉️ '123'
    console.log(price);
    
    let price1;
    if (price !== '') {
      price1 = Number(price); 
    }
    
    const rangearr = range.split("to");
    const str1 = rangearr[0].replace(/\D/g, '');
    const str2 = rangearr[1].replace(/\D/g, '');
    let range1,range2;
    if (str1 !== '') {
        range1 = Number(str1); 
      }
    if (str2 !== '') {
        range2 = Number(str2);
    }

    
    if(price1>= range1 && price1<=range2){
        return true;
    } else {
        return false;
    }

}

module.exports = {pricerange};