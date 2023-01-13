import { Component, OnInit } from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { timer } from 'rxjs';
import { fadeAnimation } from './shared/fadeAnimation';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  animations: [fadeAnimation],
})
export class AppComponent implements OnInit {
  constructor(private router: Router) {}
  ngOnInit() {
    this.router.events.subscribe((evt) => {
      if (!(evt instanceof NavigationEnd)) {
        return;
      }
      timer(280).subscribe(() => {
        window.scrollTo(0, 0);
      });
    });
  }
  getRouterOutletState(outlet) {
    return outlet.isActivated ? outlet.activatedRoute : '';
  }
}
