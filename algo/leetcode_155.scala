class MinStack() {
  import scala.collection.mutable
  import java.util.EmptyStackException

  val minElems = mutable.ListBuffer.empty[Int]
  val ordElems = mutable.ListBuffer.empty[Int]

  def push(x: Int) = {
    ordElems += x
    if (minElems.isEmpty) {
      minElems += x
    } else if (minElems.last >= x) {
      minElems += x
    }
  }

  def pop() = {
    require(minElems.nonEmpty && ordElems.nonEmpty)
    if (minElems.last == ordElems.last)
      minElems.trimEnd(1)
    ordElems.trimEnd(1)
  }

  def top() : Int = {
    ordElems.lastOption.getOrElse {
      throw new EmptyStackException()
    }
  }

  def getMin() : Int = {
    minElems.lastOption.getOrElse {
      throw new EmptyStackException()
    }
  }
}

val minStack = new MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
println(minStack.getMin())
minStack.pop()
println(minStack.top())
println(minStack.getMin())
