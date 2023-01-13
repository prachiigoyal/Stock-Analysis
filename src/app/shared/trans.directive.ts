import { Directive, ElementRef, Input, OnInit } from '@angular/core';

@Directive({
  selector: '[appTrans]',
})
export class TransDirective implements OnInit {
  @Input() transId: string;

  constructor(private elem: ElementRef) {}
  ngOnInit() {
    // this.elem.nativeElement.innerText = '.....';
  }
}
