import { Component, HostListener, OnInit } from '@angular/core';
import Typewriter from 't-writer.js'

@Component({
  selector: 'app-banner-head',
  templateUrl: './banner-head.component.html',
  styleUrls: ['./banner-head.component.css']
})
export class BannerHeadComponent implements OnInit {

  pageYOffset = 0;

  constructor() { }
  ngOnInit(): void {
    const target = document.querySelector('.tw')
    const writer = new Typewriter(target, {
      loop: true,
      typeColor: 'black'
    })

    writer
      .strings(
        400,
        "Haaruitval?",
        "Acne?",
        "Erectiestoornissen?",
        "Wij helpen je."
      )
      .start()
  }
  @HostListener('window:scroll', ['$event']) onScrollEvent($event) {
    this.pageYOffset = window.pageYOffset;
  }
}
