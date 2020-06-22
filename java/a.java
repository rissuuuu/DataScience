
class a{
  String sname;
  int sage;
  a(){
    this("Rishav");
  }
  a(String name){
    this(name,70);
  }
  a(String name,int age){
    this.sname=name;
    this.sage=age;
  }
  void disp(){
    System.out.println(sname+" "+sage);
  }
  public static void main(String[] args){
    a st=new a();
    st.disp();
  }
}
